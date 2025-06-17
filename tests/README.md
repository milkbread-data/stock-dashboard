# Stock Dashboard - Testing

This directory contains the test suite for the Stock Dashboard application.

## Running Tests

### Prerequisites

1. Install the test dependencies:
   ```bash
   pip install -r requirements-test.txt
   ```

### Running the Test Suite

To run all tests:

```bash
pytest
```

To run tests with coverage report:

```bash
pytest --cov=app --cov-report=term-missing
```

To generate an HTML coverage report:

```bash
pytest --cov=app --cov-report=html
```
The HTML report will be generated in the `htmlcov` directory. Open `htmlcov/index.html` in a web browser to view the coverage report.

### Running Specific Tests

To run a specific test file:

```bash
pytest tests/test_app.py
```

To run a specific test function:

```bash
pytest tests/test_app.py::test_is_valid_ticker -v
```

### Test Organization

- `conftest.py`: Contains test fixtures and setup code
- `test_app.py`: Contains unit tests for the application

## Writing Tests

When adding new functionality, please add corresponding tests. Follow these guidelines:

1. Test one thing per test function
2. Use descriptive test function names
3. Use fixtures for common test data
4. Mock external dependencies
5. Aim for high test coverage (80%+)

## Test Coverage

Current test coverage can be viewed by running:

```bash
pytest --cov=app --cov-report=html
```

Then open `htmlcov/index.html` in a web browser.

## Debugging Tests

To drop into the Python debugger on test failures:

```bash
pytest --pdb
```

For more verbose output:

```bash
pytest -v
```
