from jaxtyping import install_import_hook


# decorate @jaxtyped and @beartype.beartype
with install_import_hook("ppo_brax_equinox", "beartype.beartype"):
    # Any module imported inside this `with` block, whose name begins with the specified string,
    # will automatically have both `@jaxtyped` and the specified
    # typechecker applied to all of their functions.
    # print("Installing import hook for jaxtyping and beartype")
    from .wrappers import *
    # type checking still happens in the dataclasses file even if it is not directly imported.
    # and it is not because of the wildcard import before it
    from .dataclasses import *

    from .ppo import *
    from .models import *
    from .utils import *
    from .evaluator import *
