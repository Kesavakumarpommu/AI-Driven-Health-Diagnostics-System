from flask import Blueprint, render_template, redirect, url_for, session

views = Blueprint('views', __name__)

# Redirect root to auth home (login check handled there)
@views.route("/")
def home_redirect():
    return redirect(url_for('auth.home'))


# --- KIDNEY ---
@views.route("/kidney")
def kidney():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('kidney_index.html')

@views.route("/kidney_form")
def kidney_form():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('kidney.html')


# --- LIVER ---
@views.route("/liver")
def liver():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('liver_index.html')

@views.route("/liver_form")
def liver_form():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('liver.html')


# --- HEART ---
@views.route("/heart")
def heart():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('heart_index.html')

@views.route("/heart_form")
def heart_form():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('heart.html')


# --- STROKE ---
@views.route("/stroke")
def stroke():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('stroke_index.html')

@views.route("/stroke_form")
def stroke_form():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('stroke.html')


# --- DIABETES ---
@views.route("/diabete")
def diabete():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('diabete_index.html')

@views.route("/diabete_form")
def diabete_form():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('diabetes.html')
