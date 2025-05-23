import jsonargparse


def make():
    parser = jsonargparse.ArgumentParser()

    parser.add_argument(
        "-config", "--config", type=str, help="path to YAML config file"
    )
    parser.add_argument("-name", "--name", type=str, help="name of this run")
    parser.add_argument("-seed", "--general.seed", type=int, help="random seed")
    parser.add_argument("-device", "--general.device", type=int, help="GPU device id")
    parser.add_argument(
        "-wandb",
        "--use_wandb",
        action="store_true",
        default=False,
        help="log to wandb?",
    )
    # parser.add_argument(
    #     "-no_eval",
    #     "--algorithm.no_eval",
    #     action="store_true",
    #     default=False,
    #     help="no evaluation",
    # )
    parser.add_argument(
        "-val_splits",
        "--evaluation.val_splits",
        type=list,
        default=["val_sim", "val_true"],
        help="validation splits",
    )
    parser.add_argument(
        "-log_freq", "--algorithm.log_freq", type=int, help="frequency of logging"
    )
    # parser.add_argument(
    #     "-clip_vloss", "--algorithm.clip_vloss", type=int, help="Clip value loss (RL)"
    # )
    # parser.add_argument(
    #     "-norm_adv", "--algorithm.norm_adv", type=int, help="Normalize advantage (RL)"
    # )

    parser.add_argument(
        "-overwrite",
        "--overwrite",
        action="store_true",
        help="overwrite experiment folder (if exists)",
    )
    parser.add_argument(
        "-env",
        "--environment.common.env_name",
        type=str,
        help="name of the environment",
    )
    parser.add_argument(
        "-query_cost",
        "--coord_env.expert_query_cost_ratio",
        type=float,
        help="Cost of querying expert agent",
    )
    parser.add_argument(
        "-switch_cost",
        "--coord_env.switch_agent_cost_ratio",
        type=float,
        help="Cost of switching agent",
    )

    # agents
    parser.add_argument(
        "-sim_novice",
        "--agents.sim_novice",
        type=str,
        help="path to the sim novice agent",
    )
    parser.add_argument(
        "-novice", "--agents.novice", type=str, help="path to the novice agent"
    )
    parser.add_argument(
        "-expert", "--agents.expert", type=str, help="path to the expert agent"
    )

    parser.add_argument(
        "-f_n", "--file_name", type=str, help="file name for evaluation"
    )
    # parser.add_argument(
    #     "-agent",
    #     "--general.agent",
    #     type=str,
    #     choices=["novice", "expert"],
    #     help="agent to evaluate",
    # )
    parser.add_argument(
        "-cp_feature",
        "--coord_policy.feature_type",
        type=str,
        choices=[
            "obs",
            "hidden",
            "hidden_obs",
            "dist",
            "hidden_dist",
            "obs_dist",
            "obs_hidden_dist",
        ],
        help="Type of features for coordination policy",
    )
    parser.add_argument(
        "-cp_data_agent",
        "--coord_policy.collect_data_agent",
        type=str,
        choices=["novice", "expert"],
        default="novice",
        help="agent to collect data",
    )

    # always policy
    parser.add_argument(
        "-cp_agent",
        "--coord_policy.agent",
        type=str,
        choices=["novice", "expert"],
        help="always choose action of this agent",
    )

    # threshold policy
    parser.add_argument(
        "-cp_metric",
        "--coord_policy.metric",
        type=str,
        choices=["max_logit", "max_prob", "margin", "neg_entropy", "neg_energy"],
        help="metric for computing scores",
    )

    # ood policy
    parser.add_argument(
        "-cp_method",
        "--coord_policy.method",
        type=str,
        choices=["DeepSVDD"],
        help="method for detecting OOD samples",
    )

    # # random baseline policy
    # parser.add_argument(
    #     "-cp_base",
    #     "--coord_policy.baseline",
    #     action="store_true",
    #     help="baseline policy with random action 0.5 probability",
    # )

    # minigrid
    parser.add_argument(
        "-en_tr_suffix",
        "--environment.train.env_name_suffix",
        type=str,
        help="suffix for the train environment name",
    )
    parser.add_argument(
        "-en_te_suffix",
        "--environment.test.env_name_suffix",
        type=str,
        help="suffix for the test environment name",
    )

    # procgen
    # parser.add_argument(
    #     "-use_bg",
    #     "--environment.common.use_background",
    #     type=bool,
    #     default=True,
    #     help="use background - only for procgen envs",
    # )
    # parser.add_argument(
    #     "-use_mono_asset",
    #     "--environment.common.use_monochrome_assets",
    #     type=bool,
    #     default=False,
    #     help="use monochrome assets - only for procgen envs",
    # )
    # parser.add_argument(
    #     "-res_theme",
    #     "--environment.common.restrict_themes",
    #     type=bool,
    #     default=False,
    #     help="restrict themes - only for procgen envs",
    # )

    args = parser.parse_args()

    return args
