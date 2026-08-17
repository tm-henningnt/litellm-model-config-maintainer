# The proxy image: stock litellm plus this instance's two source patches.
#
# Build from the repository root, not from this directory:
#
#   docker build -f docker/Dockerfile -t litellm-patched:1.97.0 .
#
# The patches are applied at build time and verified in the same layer.
# A build fails when litellm moves the code the patch anchors on, so an
# image never ships without both fixes. That is stricter than the host
# install, where an upgrade removes both edits silently.
#
# The provider modules under `providers/` are NOT copied in. The
# maintainer's `deploy` writes them beside the Generated Config, and the
# compose file mounts that directory. So a provider change reaches the
# proxy without a rebuild, exactly as it does on the host.

ARG LITELLM_VERSION=1.97.0
FROM docker.litellm.ai/berriai/litellm:${LITELLM_VERSION}

USER root

# Keep the script in the image. `docker compose exec proxy python
# /usr/local/bin/apply_litellm_patches.py --verify` then answers the same
# question `doctor` answers on the host.
COPY docker/apply_litellm_patches.py /usr/local/bin/apply_litellm_patches.py

RUN python /usr/local/bin/apply_litellm_patches.py
