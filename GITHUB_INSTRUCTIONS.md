# GitHub Repository Setup Instructions

Follow these steps to share your Stock Dashboard app on GitHub:

## 1. Create a New Repository on GitHub

1. Go to [GitHub](https://github.com/) and sign in to your account
2. Click the "+" icon in the top-right corner and select "New repository"
3. Enter "stock-dashboard" as the repository name
4. Add a description: "Interactive Stock Market Dashboard with Yahoo Finance data"
5. Choose "Public" visibility
6. Do NOT initialize with a README, .gitignore, or license (we already have these files)
7. Click "Create repository"

## 2. Push Your Code to GitHub

After creating the repository, GitHub will show you commands to push your existing repository. Use the following commands:

```bash
# Navigate to your project directory (if you're not already there)
cd /path/to/stock-dashboard

# Add the remote repository URL
git remote add origin https://github.com/YOUR_USERNAME/stock-dashboard.git

# Push your code to GitHub
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## 3. Verify Your Repository

1. Refresh your GitHub repository page
2. You should see all your files and the README displayed

## 4. Share Your Repository

You can now share your repository URL with others:
```
https://github.com/YOUR_USERNAME/stock-dashboard
```

## 5. Deploy to Heroku (Optional)

To deploy your app to Heroku:

1. Create a Heroku account if you don't have one
2. Install the Heroku CLI
3. Run these commands:

```bash
# Login to Heroku
heroku login

# Create a new Heroku app
heroku create your-stock-dashboard

# Push to Heroku
git push heroku main

# Open your app
heroku open
```

## Note

The repository already includes all necessary files for deployment:
- `requirements.txt` - Lists all dependencies
- `Procfile` - Specifies the command to run on Heroku
- `runtime.txt` - Specifies Python version
- Server variable in app.py for deployment