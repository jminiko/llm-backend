from flask import Flask, render_template, request


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/cv'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'QmSOeEOdiDIAwqQNkSKqSJDEIOQSKslskDslk'  # Change this to a secure secret key
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)
app.secret_key =  'MyVerySecretKeyQsQmslDopPSlEZqJ'

app.config['RAGTIME_COMPS_PER_PAGE'] = 10

# Sample data for demonstration
sample_data = [
    {"id": 1, "name": "Python Programming", "description": "Learn Python programming language"},
    {"id": 2, "name": "Flask Framework", "description": "Web development with Flask"},
    {"id": 3, "name": "Database Design", "description": "Learn database design principles"},
    {"id": 4, "name": "API Development", "description": "Create RESTful APIs with Flask"},
    {"id": 5, "name": "Web Security", "description": "Security best practices for web applications"}
]

@app.route('/', methods=['GET'])
def index():
    search_query = request.args.get('query', '')
    
    if search_query:
        # Filter data based on search query
        results = [item for item in sample_data if search_query.lower() in item['name'].lower() 
                  or search_query.lower() in item['description'].lower()]
    else:
        results = sample_data
    
    return render_template('index.html', results=results, search_query=search_query)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


    