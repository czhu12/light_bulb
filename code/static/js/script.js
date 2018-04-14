$(document).ready(() => {
  $(document).keypress((e) => {
    let yes = 106;
    let no = 107;
    if (e.which == yes) {
      submitJudgement('YES')
    }
    if (e.which == no) {
      submitJudgement('NO')
    }
  });

  let itemsToLabel = [];
  let currentIndex = 0;
  let showItem = undefined
  if (window.dataType === 'images') {
    $("#image-classification").css("display", "block")
    showItem = () => {
      let item = itemsToLabel[currentIndex];
      let path = item['path']
      $("#item-image").attr("src", "/images?image_path=" + path)
    }
  } else {
    $("#text-classification").css("display", "block")
    showItem = () => {
      let item = itemsToLabel[currentIndex]
      let text = item['text']
      $("#text-classification-text").html(text)
    }
  }

  let currentItem = () => {
    return itemsToLabel[currentIndex];
  }

  let getNextBatch = () => {
    $.get('/batch', (data) => {
      let batch = data['batch'];
      itemsToLabel = itemsToLabel.concat(batch)
      if (currentIndex == 0) {
        showItem()
      }
    })
  }

  let submitJudgement = (label) => {
    let image = currentItem();
    let path = image.path;

    $.post('/judgements', {id: path, label: label}, (response) => {
      showNextItem();
      if (currentIndex + 3 > itemsToLabel.length) {
        getNextBatch();
      }
    })
  }

  let showNextItem = () => {
    currentIndex += 1
    showItem()
  }

  // Kick everything off
  getNextBatch();

  $(".judgement-button").click((el) => {
    let label = $(el.target).data("judgement")
    submitJudgement(label)
  });
})
