# Contributing to Space Weather Sentinel

We welcome contributions to the Space Weather Sentinel project! Please follow these steps to contribute:

## 1. Local Development
1. Fork the repository and clone it locally.
2. Install dependencies: `pip install -r requirements.txt`.
3. Install development dependencies: `pip install black isort flake8 pytest`.

## 2. Code Standards
All Python code must adhere to PEP-8.
We use `black` for formatting and `isort` for import sorting.
Before submitting a Pull Request, run:
```bash
make format
make lint
make test
```

## 3. Submitting a Pull Request
- Create a feature branch (`git checkout -b feature/your-feature-name`).
- Commit your changes with clear, descriptive messages.
- Push your branch and open a Pull Request.
- Ensure the GitHub Actions CI pipeline passes successfully.

Thank you for helping improve space weather forecasting!
