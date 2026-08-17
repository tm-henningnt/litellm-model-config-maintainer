"""Provider handler modules deployed next to the running proxy.

Each module here is deployed as a standalone file
(`litellm_maintainer.deploy.deploy_provider_modules` copies it flat, by
content, into the proxy's config directory). The proxy imports each one
by its bare file name (`cline_provider.cline_llm`,
`chatgpt_role_fix.chatgpt_system_role_fix`), never through this
package's dotted path. This `__init__.py` exists only so the installed
wheel carries the directory (see `pyproject.toml`,
`[tool.setuptools.packages.find]`); it is never imported as
`providers.cline_provider`.
"""
