from rlsrl.api.environment import register

register("atari", "AtariEnvironment", "rlsrl.legacy.environment.atari.atari_env")
register("football", "FootballEnvironment", "rlsrl.legacy.environment.google_football.gfootball_env")
register('gym_mujoco', "GymMuJoCoEnvironment", "rlsrl.legacy.environment.gym_mujoco.gym_mujoco_env")
register("hide_and_seek", "HideAndSeekEnvironment", "rlsrl.legacy.environment.hide_and_seek.hns_env")
register("smac", "SMACEnvironment", "rlsrl.legacy.environment.smac.smac_env")

