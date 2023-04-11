from flask import Flask, render_template, request
import movie_recommend

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        search_query = str(request.form.get("search"))
        # Use the search query to generate recommendations
        recommendations = generate_recommendations(search_query)
        return render_template("index.html", recommendations=generate_recommendations(search_query))
    else:
        return render_template("index.html")


def generate_recommendations(search_query):
    # Use the search query to generate recommendations
    # Code for generating recommendations goes here
    recommendations = movie_recommend.recommend(search_query)
    return recommendations


if __name__ == "__main__":
    app.run(debug=True)
