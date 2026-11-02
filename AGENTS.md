# Project instructions

- Keep dependencies, models, caches, and temporary files inside `.tools/`. Never change global configs.
- Use `./setup.sh` to install and `./run.sh` to process `in/` into `out/`. No persistent servers.
- Preserve inputs. Keep CtrlRegen at strength `0.25` and `50` steps unless explicitly asked otherwise.
- Local spectral scores proved unreliable; processing success is not detector verification.
- Keep scripts and documentation simple and concise. Use shieldcn badges in the README.
