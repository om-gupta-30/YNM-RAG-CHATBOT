# Contributing to RAG PDF Chatbot

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YNM-RAG-CHATBOT.git
   cd YNM-RAG-CHATBOT
   ```
3. **Set up the development environment:**
   ```bash
   make setup-env
   # Edit .env and add your GEMINI_API_KEY
   make install
   ```
4. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. **Make your changes** in your feature branch
2. **Test your changes:**
   ```bash
   make dev              # Test full application
   make lint             # Check frontend code
   make lint-backend     # Check Python code (if configured)
   ```
3. **Commit with clear messages:**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```
4. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** on GitHub

## Code Style

### Python
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings for functions and classes
- Keep functions focused and modular

### JavaScript/React
- Use modern ES6+ syntax
- Follow React best practices
- Use functional components with hooks
- Keep components small and reusable

## Commit Message Guidelines

Use clear, descriptive commit messages:

- `Add: new feature or file`
- `Update: enhancement to existing feature`
- `Fix: bug fix`
- `Refactor: code restructuring`
- `Docs: documentation changes`
- `Style: formatting, no code change`
- `Test: adding or updating tests`

## Pull Request Process

1. Ensure your code follows the project's style guidelines
2. Update the README.md if you're adding new features
3. Add comments to explain complex logic
4. Test thoroughly before submitting
5. Reference any related issues in your PR description

## Security

- **NEVER commit API keys or secrets**
- Always use environment variables for sensitive data
- Review the [SECURITY.md](SECURITY.md) file
- Report security vulnerabilities privately (see SECURITY.md)

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Suggestions for improvements

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what's best for the project and community
- Accept constructive criticism gracefully

Thank you for contributing! 🚀
