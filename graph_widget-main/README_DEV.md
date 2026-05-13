# Development information
## Pushing to pypi
- git commit and push
- empty the `dist` directory
- In `pyproject.toml` update version.
- `uv build`
- `uv publish`