# Security Policy

## 🔒 Protecting Your API Keys

### ⚠️ IMPORTANT: Never Commit API Keys to Git

This project requires API keys for LLM providers. **NEVER** commit your actual API keys to the repository.

### ✅ Safe Practices

1. **Use Environment Variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your actual keys
   nano .env
   ```

2. **The `.env` file is in `.gitignore`**
   - Your `.env` file will NOT be committed to git
   - Only `.env.example` (with placeholder values) is tracked

3. **Check Before Committing**
   ```bash
   # Always check what you're committing
   git status
   git diff
   
   # Make sure no API keys are visible
   git grep -i "sk-" || echo "No OpenAI keys found"
   git grep -i "api[_-]key.*=.*[a-z0-9]" || echo "No hardcoded keys found"
   ```

### 📋 Protected Files (in `.gitignore`)

The following patterns are automatically excluded from git:

- `.env`
- `.env.local`
- `.env.*.local`
- `**/api_keys.txt`
- `**/*_secret.rs`
- `**/*secret*`
- `.longcodezip/` (cache directory)
- `relevance_cache.json`

### 🚨 What to Do If You Accidentally Commit a Key

1. **Immediately revoke the exposed key** at your provider's dashboard
2. **Remove from git history:**
   ```bash
   # Remove the file from git history
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all
   
   # Force push (WARNING: coordinate with team first)
   git push origin --force --all
   ```

3. **Generate a new API key**
4. **Update your local `.env` file** with the new key

### 🛡️ Additional Security Measures

#### For Examples and Tests

All example files use safe patterns:

```rust
// ✅ GOOD: Use environment variables
let api_key = env::var("DEEPSEEK_API_KEY")
    .unwrap_or_else(|_| "api-key".to_string());

// ✅ GOOD: Use placeholder values
let provider = ProviderConfig::deepseek("your-api-key");

// ❌ BAD: Never hardcode real keys
let provider = ProviderConfig::deepseek("sk-abc123..."); // DON'T DO THIS!
```

#### Cache Files

The cache may contain API response data but NOT API keys:
- Cache is stored in `~/.longcodezip/cache/` by default
- Cache files are also in `.gitignore`
- Cache is automatically cleaned based on TTL

### 📝 Reporting Security Issues

If you discover a security vulnerability, please email:
- **Email**: security@example.com (replace with actual contact)
- **Subject**: [SECURITY] LongCodeZip-rs vulnerability

Please do NOT open a public issue for security vulnerabilities.

### 🔍 Regular Security Checks

Before each commit, run:

```bash
# Check for potential secrets
git diff --cached | grep -i "api.*key\|secret\|password\|token"

# Use git-secrets (optional but recommended)
git secrets --scan
```

### 🎯 Using Local Providers (No API Keys Needed)

For maximum security, use local LLM providers that don't require API keys:

- **Ollama**: `http://localhost:11434`
- **LM Studio**: `http://localhost:1234`
- **llama.cpp**: `http://localhost:8080`

Example:
```rust
// No API key needed!
let provider = ProviderConfig::ollama("llama3.1:8b", None);
```

### 📚 Resources

- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [git-secrets tool](https://github.com/awslabs/git-secrets)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)

---

**Remember**: When in doubt, use environment variables and never commit `.env` files!
