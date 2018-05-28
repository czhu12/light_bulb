$(document).ready(() => {
  if (window.dataType == "images") {
    $("#classify-image").click((el) => {
      el.preventDefault()
      let url = $("#image-classification").val()
      $("#selected-image").attr("src", url)

      $.ajax({
        type: "POST",
        url: '/score',
        data: JSON.stringify({ urls: [url], type: window.dataType }),
        dataType: "json",
        contentType: "application/json",
        success: (response) => {
          response['scores'].map((score) => {
            if (score[0] > score[1]) {
              // negative classification
              $("#result").html(`NO (${Math.floor(100 * score[0])}%)`)
            } else {
              // positive classification
              $("#result").html(`YES (${Math.floor(100 * score[1])}%)`)
            }
          })
        }
      });
    })
  } else if (window.dataType == "text") {
    $("#classify-text").click((el) => {
      el.preventDefault()
      let text = $("#text-classification").val()

      $.ajax({
        type: "POST",
        url: '/score',
        data: JSON.stringify({ texts: [text], type: window.dataType }),
        dataType: "json",
        contentType: "application/json",
        success: (response) => {
          response['scores'].map((score) => {
            if (score[0] > score[1]) {
              // negative classification
              $("#result").html(`NO (${Math.floor(100 * score[0])}%)`)
            } else {
              // positive classification
              $("#result").html(`YES (${Math.floor(100 * score[1])}%)`)
            }
          })
        }
      });
    })
  }
})
