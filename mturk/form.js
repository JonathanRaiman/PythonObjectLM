Element.prototype.prependChild = function(child) { this.insertBefore(child, this.firstChild); };
function add_warning_message(id) {
	var el = document.getElementById(id),
		warning = document.createElement('p');
	warning.innerHTML = "<b>You are previewing this HIT.</b> To perform this HIT, please accept it.";
	el.prependChild(warning);
}
function disable(id){document.getElementById(id).disabled=true;}
function hide(id){document.getElementById(id).style.visibility="hidden";}
function populateAssignmentID(field_id) {
	var parampairs = window.location.search.substring(1).split("&");
	for (var i in parampairs) {
		var pair = parampairs[i].split("=");
		if (pair[0] == "assignmentId") {
			if (pair[1] == "ASSIGNMENT_ID_NOT_AVAILABLE") {
				add_warning_message("unsure_if_answerable");
				add_warning_message("overview_box");
				hide("submit_answer");
				disable("explanation_select");
				disable("explanation_comments");
				disable("recommendation_select");
			} else {document.getElementById(field_id).value=pair[1];}
			return;
		}
	}
}