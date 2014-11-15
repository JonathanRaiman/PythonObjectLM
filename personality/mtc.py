import datetime
from boto.mturk.question import HTMLQuestion 
from warnings import warn

def convert_to_select_options(options):
    option_list = ""
    for option in options:
        option_list += """<option value="{option_value}">{option_name}</option>""".format(
                option_value=option,
                option_name=option)
    return option_list

def convert_to_input_list(example_names):
    return "'" + "', '".join(example_names) + "'"

def create_mtc_question(mtc,
    exampleset,
    duration = datetime.timedelta(0,1500),
    reward = 0.5,
    max_assignments = 3,
    title = 'Help recommend a restaurant to a friend.'):

    if hasattr(exampleset, "HITId"):
        warn("ExampleSet was already submitted")
        return None
    else:

        description = ('Read short snippets about places someone liked'
                       '  or disliked, and choose among a list which corresponds best')
        keywords = 'rating, opinions, recommendation'
        url = "https://workersandbox.mturk.com/mturk/externalSubmit" if ("sandbox" in mtc.host) else "https://www.mturk.com/mturk/externalSubmit"
        form = open("mturk/form.html").read().format(
            style_code = open("mturk/form.css").read(),
            javascript=open("mturk/form.js").read(),
            difficulty = exampleset.difficulty,
            url = url,
            input_restaurants = convert_to_input_list(exampleset.example_names()),
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

        exampleset.HITId = hit[0].HITId
        return hit

def get_responses_for_hit(mtc, HITId):
    hit_response = []
    assignments = mtc.get_assignments(HITId)
    for assignment in assignments:
        # worker = assignment.WorkerId
        hit_worker_response = {question_form_answer.qid: question_form_answer.fields[0] for answer in assignment.answers for question_form_answer in answer if question_form_answer.qid != "submit"}
        hit_worker_response["HITId"] = HITId
        hit_response.append(hit_worker_response)
    return hit_response

def collect_hit_responses(mtc):
    hits = get_all_reviewable_hits(mtc, verbose=False)
    responses = []
    for hit in hits:
        responses.append(get_responses_for_hit(mtc, hit.HITId))
    return responses

def get_all_reviewable_hits(mtc, verbose = True):
    page_size = 50
    hits = mtc.get_reviewable_hits(page_size=page_size)
    if verbose: print("Total results to fetch %s " % hits.TotalNumResults)
    if verbose: print("Request hits page %i" % 1)
    total_pages = float(hits.TotalNumResults)/page_size
    int_total= int(total_pages)
    if(total_pages-int_total>0):
        total_pages = int_total+1
    else:
        total_pages = int_total
    pn = 1
    while pn < total_pages:
        pn = pn + 1
        if verbose: print("Request hits page %i" % pn)
        temp_hits = mtc.get_reviewable_hits(page_size=page_size,page_number=pn)
        hits.extend(temp_hits)
    return hits