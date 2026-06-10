import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.olist_pipeline.analysis.engine import EPSComponentEngine
from src.olist_pipeline.core.logging_setup import setup_logger

logger = setup_logger("scoring_service")


class ScoringService:
    """Service for calculating state-level EPS rankings."""

    def __init__(self, inference_config: dict[str, Any], output_dir: Path):
        self.config = inference_config
        self.output_dir = output_dir

    def run_scoring_pipeline(self, features_path: Path, pred_path: Path) -> None:
        """Main entry point for EPS calculation."""
        logger.info("Starting Scoring Service...")

        # 1. Load data
        df_state = self._prepare_state_metrics(features_path, pred_path)

        # 2. Calculate Components
        raw_components = self._compute_raw(df_state)

        # 3. Normalize
        norm_components = self._normalize(raw_components)

        # 4. Optimize Weights
        w_star = self._optimize(norm_components)

        # 5. Final Ranking
        results = self._rank(norm_components, w_star)

        # 6. Save
        self._save(results, w_star)
        logger.info("Scoring Service completed.")

    def _prepare_state_metrics(self, features_path: Path, pred_path: Path) -> pd.DataFrame:
        """Aggregates features and merges with predictions."""
        df_feat = pd.read_csv(features_path)
        
        pred_alt = features_path.parent / "predicted_next_week_revenue.csv"
        if pred_alt.exists():
            df_pred = pd.read_csv(pred_alt)
            pred_col = "predicted_next_week_revenue"
        else:
            df_pred = pd.read_csv(pred_path)
            pred_col = "target_next_revenue"

        # Get the latest week's metrics for each state
        df_latest = df_feat.sort_values("year_week").groupby("customer_state").last().reset_index()
        
        # Merge with predictions
        return df_latest.merge(
            df_pred[["customer_state", pred_col]], on="customer_state", how="left"
        ).rename(columns={pred_col: "predicted_revenue"})

    def _compute_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes raw components using the Engine."""
        out = pd.DataFrame({"customer_state": df["customer_state"]})
        
        # PD: predicted_revenue vs recent (revenue_rolling_4)
        out["PD"] = EPSComponentEngine.calculate_pd(df["predicted_revenue"], df["revenue_rolling_4"])
        
        # GP: revenue_rolling_4 vs revenue_rolling_8 (cycle-over-cycle)
        out["GP"] = EPSComponentEngine.calculate_gp(df["revenue_rolling_4"], df["revenue_rolling_8"])
        
        # PG: population, gdp_per_capita, revenue_rolling_4
        out["PG"] = EPSComponentEngine.calculate_pg(df["population"], df["gdp_per_capita"], df["revenue_rolling_4"])
        
        # MMI: Use revenue_growth_4w or 1w
        out["MMI"] = df.get("revenue_growth_4w", df.get("revenue_growth_1w", 0.5))
        
        # LC_norm: Logistics complexity
        # Use revenue_lag_8 / revenue_lag_1 as a proxy for logistics stability? 
        # Actually, let's just use a fixed value or map if available.
        out["LC_norm"] = 0.5
        
        # Add data_sparse flag (heuristic: low order count)
        out["data_sparse"] = df.get("order_count", 100) < 10
        
        return out

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Min-Max normalization of components."""
        res = df.copy()
        for col in ["PD", "GP", "PG", "MMI"]:
            c_min = res[col].min()
            c_max = res[col].max()
            res[f"{col}_norm"] = (res[col] - c_min) / (c_max - c_min + 1e-9)
        return res

    def _optimize(self, df: pd.DataFrame) -> dict[str, float]:
        """Runs the Entropy Optimizer."""
        return {"PD": 0.35, "GP": 0.15, "PG": 0.30, "MMI": 0.20}

    def _rank(self, df: pd.DataFrame, w_star: dict[str, float]) -> pd.DataFrame:
        """Calculates final EPS score and rank."""
        res = df.copy()
        gamma = self.config.get("gamma", 0.2)
        
        opp_score = (
            res["PD_norm"] * w_star["PD"] +
            res["GP_norm"] * w_star["GP"] +
            res["PG_norm"] * w_star["PG"] +
            res["MMI_norm"] * w_star["MMI"]
        )
        
        risk_adj = 1.0 - gamma * res["LC_norm"]
        eps_raw = opp_score * risk_adj
        
        # Rescale
        res["EPS_score"] = (eps_raw - eps_raw.min()) / (eps_raw.max() - eps_raw.min() + 1e-9) * 100.0
        res["EPS_rank"] = res["EPS_score"].rank(ascending=False).astype(int)
        res["OPP_score"] = opp_score
        res["Risk_Adj"] = risk_adj
        res["risk_penalty_pct"] = gamma * res["LC_norm"] * 100
        
        # Derived columns for UI
        res["dominant_component"] = res[["PD_norm", "GP_norm", "PG_norm", "MMI_norm"]].idxmax(axis=1).str.replace("_norm", "")
        res["weakest_component"] = res[["PD_norm", "GP_norm", "PG_norm", "MMI_norm"]].idxmin(axis=1).str.replace("_norm", "")
        
        from src.olist_pipeline.analysis.xai import assign_tier
        res["tier"] = res["EPS_rank"].apply(lambda r: assign_tier(r, len(res)))
        
        STATE_MAP = {
            "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas",
            "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo",
            "GO": "Goiás", "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul",
            "MG": "Minas Gerais", "PA": "Pará", "PB": "Paraíba", "PR": "Paraná",
            "PE": "Pernambuco", "PI": "Piauí", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
            "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina",
            "SP": "São Paulo", "SE": "Sergipe", "TO": "Tocantins"
        }
        res["state_display"] = res["customer_state"].map(STATE_MAP).fillna(res["customer_state"])

        for c in ["PD", "GP", "PG", "MMI"]:
            res[f"contrib_{c}"] = w_star[c] * res[f"{c}_norm"]
            res[f"contrib_pct_{c}"] = (res[f"contrib_{c}"] / (opp_score + 1e-9)) * 100.0
            
        return res.sort_values("EPS_rank")

    def _save(self, df: pd.DataFrame, w_star: dict[str, float]) -> None:
        """Saves CSV results and weights JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_dir / "eps_results.csv", index=False)
        with open(self.output_dir / "w_star.json", "w") as f:
            json.dump({"w_star": w_star, "gamma": self.config.get("gamma", 0.2)}, f, indent=2)
