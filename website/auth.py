from flask import Blueprint, render_template, request, redirect, url_for, session, flash, current_app
from werkzeug.security import generate_password_hash, check_password_hash

auth = Blueprint('auth', __name__)

@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        users = current_app.mongo_db.users
        data = request.form

        # Check if email already exists
        if users.find_one({'email': data['email']}):
            flash("Email already registered", "error")
            return redirect(url_for('auth.signup'))

        # Check if passwords match
        if data['password'] != data['confirm_password']:
            flash("Passwords do not match", "error")
            return redirect(url_for('auth.signup'))

        # Hash password and store user
        hashed_pw = generate_password_hash(data['password'])
        users.insert_one({
            'first_name': data['first_name'],
            'last_name': data['last_name'],
            'email': data['email'],
            'password': hashed_pw
        })

        flash("Signup successful! Please login.", "success")
        return redirect(url_for('auth.login'))

    return render_template('signup.html')


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = current_app.mongo_db.users
        data = request.form
        user = users.find_one({'email': data['email']})

        if user and check_password_hash(user['password'], data['password']):
            session['user'] = user['first_name']  # store user’s first name
            flash(f"Welcome back, {user['first_name']}!", "success")
            return redirect(url_for('auth.home'))  # go to home page

        flash("Invalid credentials", "error")
        return redirect(url_for('auth.login'))

    return render_template('login.html')


@auth.route('/')
@auth.route('/home')
def home():
    user = session.get('user')

    if user:
        return render_template('base.html', user=user)
    else:
        flash("Please log in first.", "info")
        return redirect(url_for('auth.login'))


@auth.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('auth.login'))
