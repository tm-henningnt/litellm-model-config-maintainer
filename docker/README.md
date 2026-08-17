# The proxies in containers

The main proxy, the ChatGPT seat workers and the database run as
containers. The maintainer stays on macOS.

## Why the maintainer stays on the Mac

Headroom comes from `codexbar`, a Mach-O binary that reads in-app
provider toggles. It cannot run on Linux. The tick and the headroom
refresh therefore stay under launchd, unchanged.

This costs nothing. The maintainer writes the Generated Config to
`~/.config/litellm/config.yaml`, the compose file mounts that directory,
and the proxy reloads on the write.

## What the image adds

The image is stock litellm plus this instance's two source patches.
`docker/apply_litellm_patches.py` applies them during the build and
verifies the markers in the same layer. A build fails when litellm moves
the code a patch anchors on.

This is stricter than the host install. There, `uv tool upgrade litellm`
removes both edits and nothing reports the loss.

The provider modules are not copied into the image. `deploy` writes them
beside the Generated Config, and the compose file mounts that directory.
A provider change reaches the proxy with a restart, not a rebuild.

## One hostname, both sides

The maintainer's prober calls a Declared Offering's `api_base` directly
from the Mac (`prober.py`, `base_url`). The main proxy calls the same
URL from inside the container network. The two must agree.

Measured 2026-08-17 on OrbStack:

| Name | From the Mac | From a container |
| --- | --- | --- |
| `chatgpt-seat-1` | fails | 200 |
| `chatgpt-seat-1.orb.local`, no alias | 200 | fails |
| `chatgpt-seat-1.orb.local` as a network alias | 200 | 200 |

Two separate mechanisms produce that last row, and both are needed.

OrbStack serves `<container_name>.orb.local` to the Mac. So
`container_name` must be `chatgpt-seat-1`, not a decorated form. A
container named `litellm-chatgpt-seat-1` answers only on
`litellm-chatgpt-seat-1.orb.local`, and the Mac then fails to reach the
name Policy states.

Docker's own DNS serves the network alias inside the network. So each
seat also declares `chatgpt-seat-1.orb.local` as an alias.

Keep both. Dropping either one breaks one side, and the maintainer reads
a failed ChatGPT probe as a dead seat.

## Files this needs

Three files hold secrets. Each is mode 600, and none is in this
repository.

| File | Holds |
| --- | --- |
| `~/.config/litellm/.env` | every provider credential, `LITELLM_MASTER_KEY`, `DATABASE_URL` |
| `~/.config/litellm/db.env` | `POSTGRES_PASSWORD` |
| `~/.config/litellm/seat-1.env`, `seat-2.env` | `LITELLM_WORKER_KEY` per seat |

`db.env` is separate because compose needs `POSTGRES_PASSWORD` at parse
time, and because the database has no reason to read a provider key.

## Build

Build from the repository root:

```
docker build -f docker/Dockerfile -t litellm-patched:1.97.0 .
```

## Start

Warning: stop the host proxy and the host seat workers first. The host
proxy holds port 4000.

```
docker compose --env-file ~/.config/litellm/db.env -f docker/compose.yaml up -d
```

Seat 2 is off by default, because Policy comments out every seat 2
entry. Add `--profile seat-2` to start it.

## Cut over from the host processes

Do these in order. Step 3 is the only Policy change the move needs.

1. Stop the host proxy and the host seat worker.

   ```
   pkill -f 'litellm --config config.yaml'
   pkill -f 'chatgpt-worker.yaml'
   ```

2. Confirm ports 4000 and 4011 are free:

   ```
   lsof -nP -iTCP:4000 -sTCP:LISTEN; lsof -nP -iTCP:4011 -sTCP:LISTEN
   ```

3. Point Policy at the seat's container name. Edit
   `~/.config/litellm-maintainer/policy.yaml`:

   ```
   sed -i '' 's|http://127.0.0.1:4011/v1|http://chatgpt-seat-1.orb.local:4011/v1|g' \
     ~/.config/litellm-maintainer/policy.yaml
   ```

   Do this only after the containers run. The name resolves to nothing
   while they are down, and every ChatGPT probe then fails.

4. Start the stack, then regenerate the config:

   ```
   docker compose --env-file ~/.config/litellm/db.env -f docker/compose.yaml up -d
   litellm-maintainer deploy --env ~/.config/litellm/.env
   ```

To go back, stop the containers, revert the `sed` in step 3, and run
each seat's `start.sh` again.

## Verify

Confirm the patches in the running image:

```
docker compose exec proxy python /usr/local/bin/apply_litellm_patches.py --verify
```

Run this after every image rebuild. It is the only check that reads the
litellm the proxy actually runs.

`doctor` cannot see inside a container. Run from this repository it
reads the copy the `dev` extra installs into `.venv`, which is stock
litellm, so both `litellm_patch.*` checks report `[FAIL]`. Expect those
two failures while the proxy is containerized, and answer them with the
`--verify` command above.

Then confirm the proxy answers:

```
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:4000/health/liveliness
```

## Upgrade litellm

1. Change `LITELLM_VERSION` in `docker/Dockerfile`.
2. Build. A failed build means litellm moved the patched code. Read
   `docs/gotchas.md` and correct `docker/apply_litellm_patches.py`.
3. Start the new image.
4. Run the `--verify` command above.

The fastapi pin the host install needs does not apply here. The official
image ships a working fastapi and prisma.

## Database

The database keeps the volume `litellm-db-data`. The volume is declared
`external`, so `docker compose down -v` cannot drop it.

Back it up with:

```
docker exec litellm-db pg_dump -U litellm litellm > backup.sql
```

Never set `STORE_MODEL_IN_DB`. The maintainer owns the model list. That
flag creates a second source of truth for the same thing.
