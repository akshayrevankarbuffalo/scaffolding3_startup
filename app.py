"""
app.py
Flask application template for the warm-up assignment

Students need to implement the API endpoints as specified in the assignment.
"""

from flask import Flask, request, jsonify, render_template
from starter_preprocess import TextPreprocessor
import traceback

app = Flask(__name__, template_folder="templates")
preprocessor = TextPreprocessor()


@app.route('/')
def home():
    """Render a simple HTML form for URL input"""
    return render_template('index.html')


@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Text preprocessing service is running"
    })


@app.route('/api/clean', methods=['POST'])
def clean_text():
    """
    API endpoint that accepts a URL and returns cleaned text

    Expected JSON input:
        {"url": "https://www.gutenberg.org/files/1342/1342-0.txt"}

    Returns JSON:
        {
            "success": true/false,
            "cleaned_text": "...",
            "statistics": {...},
            "summary": "...",
            "error": "..." (if applicable)
        }
    """
    try:
        # Get JSON data from request
        json_payload = request.get_json(force=True)
        if not json_payload:
            return jsonify({"success": False, "error": "No JSON payload provided."}), 400

        # Extract URL from the JSON
        target_url = json_payload.get("url")
        if not target_url:
            return jsonify({"success": False, "error": "Missing 'url' field in request."}), 400

        # Validate URL (should be .txt)
        if not isinstance(target_url, str) or not target_url.strip().lower().endswith(".txt"):
            return jsonify({"success": False, "error": "URL must be a string ending in .txt."}), 400

        # Use preprocessor.fetch_from_url()
        raw_content = preprocessor.fetch_from_url(target_url)

        # Clean the text with preprocessor.clean_gutenberg_text()
        cleaned_content = preprocessor.clean_gutenberg_text(raw_content)

        # Normalize with preprocessor.normalize_text()
        normalized_content = preprocessor.normalize_text(
            cleaned_content, preserve_sentences=True
        )

        # Get statistics with preprocessor.get_text_statistics()
        text_analysis = preprocessor.get_text_statistics(normalized_content)

        # Create summary with preprocessor.create_summary()
        text_summary = preprocessor.create_summary(
            normalized_content, num_sentences=3)

        # Return JSON response
        return jsonify({
            "success": True,
            "cleaned_text": normalized_content,
            "statistics": text_analysis,
            "summary": text_summary
        })

    except Exception as e:
        traceback.print_exc()  # Log the full error to the console
        return jsonify({
            "success": False,
            "error": f"An unexpected server error occurred: {str(e)}"
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """
    API endpoint that accepts raw text and returns statistics only

    Expected JSON input:
        {"text": "Your raw text here..."}

    Returns JSON:
        {
            "success": true/false,
            "statistics": {...},
            "error": "..." (if applicable)
        }
    """
    try:
        # Get JSON data from request
        json_payload = request.get_json(force=True)
        if not json_payload:
            return jsonify({"success": False, "error": "No JSON payload provided."}), 400

        # Extract text from the JSON
        input_text = json_payload.get("text")
        if not input_text:
            return jsonify({"success": False, "error": "Missing 'text' field in request."}), 400

        # Get statistics with preprocessor.get_text_statistics()
        analysis = preprocessor.get_text_statistics(input_text)

        # Return JSON response
        return jsonify({
            "success": True,
            "statistics": analysis
        })

    except Exception as e:
        traceback.print_exc()  # Log the full error to the console
        return jsonify({
            "success": False,
            "error": f"An unexpected server error occurred: {str(e)}"
        }), 500

# Error handlers


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


if __name__ == '__main__':
    print("🚀 Starting Text Preprocessing Web Service...")
    print("📖 Available endpoints:")
    print("   GET  /            - Web interface")
    print("   GET  /health      - Health check")
    print("   POST /api/clean   - Clean text from URL")
    print("   POST /api/analyze - Analyze raw text")
    print()
    print("🌐 Open your browser to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")

    app.run(debug=True, port=5000, host='0.0.0.0')
