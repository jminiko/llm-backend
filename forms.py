from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, FileField, DateField
from wtforms.validators import DataRequired, Length, Email, EqualTo

class ApplicationForm(FlaskForm):
    query_text = TextAreaField('Recherche')
    submit = SubmitField('Afficher')