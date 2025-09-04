import jax
import jax.numpy as jnp

if __name__ == "__main__":
    data = {
        "l": jnp.array([0, 1]),
        "a": {
            "c": jnp.array([4, 5]),
        },
    }

    def fn(carry, struct):
        jax.debug.print("{}", struct)
        return carry + 1, struct

    final, acc = jax.lax.scan(fn, init=0, xs=data)

    print(f"Final is {final} and accumulation is {acc}")
