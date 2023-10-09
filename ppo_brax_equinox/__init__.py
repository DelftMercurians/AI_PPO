from jaxtyping import install_import_hook


# decorate @jaxtyped and @beartype.beartype
with install_import_hook("ppo_brax_equinox", "beartype.beartype"):
    # Any module imported inside this `with` block, whose name begins with the specified string,
    # will automatically have both `@jaxtyped` and the specified
    # typechecker applied to all of their functions.
    print("Installing import hook for jaxtyping and beartype")
    from .wrappers import ObservationNormalizingWrapper
