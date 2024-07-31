from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__, template_folder='../UI')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'email_text' in request.form:
        email_text = request.form['email_text']
        result = util.classify_email(email_text)
        return jsonify({'result': result})
    return jsonify({'error': 'No text provided'})

if __name__ == '__main__':
    app.run(debug=True)
