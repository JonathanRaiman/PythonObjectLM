import datetime
from boto.mturk.question import HTMLQuestion 

def convert_to_select_options(options):
    option_list = ""
    for option in options:
        option_list += "<option value={option_value}>{option_name}</option>".format(
                option_value=option,
                option_name=option)
    return option_list

def create_mtc_question(mtc,
    exampleset,
    duration = datetime.timedelta(0,1500),
    reward = 0.5,
    max_assignments = 3,
    title = 'Help recommend a restaurant to a friend.'):
    description = ('Read short snippets about places someone liked'
                   '  or disliked, and choose among a list which corresponds best')
    keywords = 'rating, opinions, recommendation'
    url = "https://workersandbox.mturk.com/mturk/externalSubmit" if ("sandbox" in mtc.host) else "https://www.mturk.com/mturk/externalSubmit"
    form = open("mturk/form.html").read().format(
        style_code = open("mturk/form.css").read(),
        javascript=open("mturk/form.js").read(),
        difficulty = exampleset.difficulty,
        url = url,
        input_restaurants = "'" + "', '".join(exampleset.example_names()) + "'",
        restaurant_options = convert_to_select_options(exampleset.option_names()),
        personality_type = exampleset.personality_type,
        examples = "".join(exampleset.get_examples_html()),
        options = "".join(exampleset.get_options_html())
    )
    qc1 = HTMLQuestion(html_form=form, frame_height=4500)
    
    hit = mtc.create_hit(question=qc1,
               max_assignments=max_assignments,
               title=title,
               description=description,
               keywords=keywords,
               duration = duration,
               reward=reward)
    
    return hit