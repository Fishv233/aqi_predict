import lightgbm as lgb
import wandb


def build_model(config: dict):
    # 從 wandb 獲取超參數配置

    # 創建 LightGBM 模型
    model = lgb.LGBMRegressor(
        n_estimators=config.pop("n_estimators", 100),
        max_depth=config.pop("max_depth", 3),
        learning_rate=config.pop("learning_rate", 0.1),
        subsample=config.pop("subsample", 0.8),
        colsample_bytree=config.pop("colsample_bytree", 0.8),
        min_child_weight=config.pop("min_child_weight", 1),
        reg_alpha=config.pop("reg_alpha", 0),
        reg_lambda=config.pop("reg_lambda", 0),
        random_state=42,
    )
    return model
