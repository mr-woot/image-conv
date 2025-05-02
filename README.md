# Employee Image Processing Tool

A Streamlit application for processing employee profile and signature images from ZIP archives.

## Features

- Extract and standardize employee profile and signature images
- Process nested ZIP files or direct image files
- Auto-convert PDF files to images (requires Poppler)
- Optimize images for size requirements (50KB-100KB)
- Support for large file uploads (up to 5GB)

## Installation

1. Clone the repository
2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run main.py
   ```

## Configuration for Large File Uploads

### Local Development

The application automatically configures Streamlit to handle files up to 5GB when run locally.

### Streamlit Cloud Deployment

When deploying to Streamlit Cloud, create a `.streamlit/config.toml` file with the following settings to handle large uploads:

```toml
[server]
maxUploadSize = 5000
maxMessageSize = 1000
enableXsrfProtection = false
enableCORS = false
enableWebsocketCompression = true
timeout = 3600
connectionTimeout = 3600
```

These settings:
- Allow up to 5GB file uploads (`maxUploadSize = 5000`)
- Set longer timeouts for large file processing (1 hour)
- Configure WebSocket compression for better performance

## Troubleshooting Network Errors

If you encounter "Network Error" when uploading large files:

1. **Split your ZIP files** into smaller chunks (200-300MB each)
2. **Try a different browser** (Firefox or Chrome often work best)
3. **Verify your config.toml** file is correctly set up for cloud deployment
4. **Check network stability** and avoid interruptions during upload
5. **Increase timeout duration** if uploads are timing out

For Axios Network Error specifically:
- Check that all required configuration is in place
- Verify the browser console for more detailed error information
- Consider using the application locally for very large files

## PDF Support

For PDF conversion:
1. Install Poppler:
   - Windows: Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)
   - Mac: `brew install poppler`
   - Linux: `apt-get install poppler-utils`
2. Ensure pypdf and pdf2image packages are installed:
   ```
   pip install pypdf pdf2image
   ```

## License

[MIT License](LICENSE) 