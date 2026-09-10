# PersonId Streamlit deployment

This guide covers deployment of the Streamlit dashboard to Streamlit Community Cloud.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub. The Streamlit app entrypoint is `streamlit_app.py`.
2. Keep the `known_people/` directory in the repository if the dashboard needs the enrolled gallery. Streamlit Cloud cannot read files that exist only on your local machine.
3. In Streamlit Community Cloud, choose **Deploy an app**, select this repository and branch, and set the main file to `streamlit_app.py`.
4. Keep the dependency file set to `requirements.txt`.
5. Open the app's **Settings > Secrets** panel and paste the contents of `.streamlit/secrets.toml.example`, replacing every placeholder.
6. Deploy. The app creates the required database tables on the first successful MySQL connection.

## Required secrets

Add these values in the Streamlit Cloud Secrets panel:

```toml
REID_ADMIN_PASSWORD = "replace-with-a-long-random-password"

[mysql]
host = "your-mysql-host"
port = 3306
user = "your-mysql-user"
password = "your-mysql-password"
database = "person_reid"
```

Use a MySQL database reachable from the public internet. Allowlist Streamlit Cloud's outbound access if your database provider requires it.

## Deployment checklist

- `streamlit_app.py` is selected as the app file.
- `requirements.txt` is present at the repository root.
- The `known_people/` gallery is present if required by the app.
- MySQL accepts the Cloud connection and the database user can create tables and triggers.
- `REID_ADMIN_PASSWORD` is set in Cloud Secrets.
- No real passwords or database credentials are committed to Git.

## Troubleshooting

- **MySQL is not configured:** verify the names and indentation in the Cloud Secrets panel.
- **Database connection fails:** confirm the host, port, user, password, database name, and provider allowlist.
- **No data appears:** run the local pipeline separately so it writes completed runs to the configured MySQL database.