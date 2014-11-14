function populateAssignmentID(field_id) {
	var assignment_id_field = document.getElementById(field_id);
	var paramstr = window.location.search.substring(1);
	var parampairs = paramstr.split("&");
	for (var i in parampairs) {
		var pair = parampairs[i].split("=");
		if (pair[0] == "assignmentId") {
			if (pair[1] == "ASSIGNMENT_ID_NOT_AVAILABLE") {


				var el = document.getElementById("unsure_if_answerable");
				if (el && el.innerHTML) {
					el.innerHTML =  "<p><b>You are previewing this HIT.</b>  To perform this HIT, please accept it.</p>" + el.innerHTML;
				}

				document.getElementById("submit_answer").disabled = true;
				document.getElementById("explanation_select").disabled = true;
				document.getElementById("explanation_comments").disabled = true;
				document.getElementById("recommendation_select").disabled = true;

			} else {
				assignment_id_field.value = pair[1];
			}
			return;
		}
	}
}