$(document).ready(() => {
  $(".token").click((el) => {
    let token = $(el.target).data("token")
    let val = $(".sequence-input input").val()
    // Take the tokens and join with space
    let tokens = val.split(' ')
    tokens.push(token)
    let newVal = tokens.filter((t) => { return !!t }).join(' ')
    $(".sequence-input input").val(newVal)
    $(".sequence-input input").focus()
  })

  // Start Key Handling
  $(document).keypress((e) => {
    if (window.labelType === 'binary') {
      let yes = 106
      let no = 107
      if (e.which === yes) {
        submitJudgement('YES')
      }
      if (e.which === no) {
        submitJudgement('NO')
      }
    } else if (window.labelType === 'classification') {

    } else if (window.labelType === 'sequence') {
      let enter = 13
      if (e.which === enter) {
        submitJudgement($(".sequence-input input").val())
        $(".sequence-input input").val('')
      }
    }
  })
  // End Key Handling

  let itemsToLabel = []
  let currentIndex = 0
  let showItem = undefined
  if (window.dataType === 'images') {
    $("#image-classification").css("display", "block")
    showItem = () => {
      let item = itemsToLabel[currentIndex]
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
    return itemsToLabel[currentIndex]
  }

  let getTrainingHistory = () => {
    $.get('/history', (data) => {
      let history = data['history']
      let bestAccByNumLabels = {}

      history.forEach((hist) => {
        if (!(hist['num_labels'] in bestAccByNumLabels)) {
          bestAccByNumLabels[hist['num_labels']] = hist['test']['acc']
        }

        bestAccByNumLabels[hist['num_labels']] = Math.max(
          hist['test']['acc'],
          bestAccByNumLabels[hist['num_labels']],
        )
      })

      let maxLabels = Math.max(Object.keys(bestAccByNumLabels))
      let x = []
      let y = []
      for (let i = 0; i <= maxLabels; i++) {
        x.push(i)
        if (i in bestAccByNumLabels) {
          y.push(bestAccByNumLabels[i])
        } else {
          y.push(null)
        }
      }
      setLabelledCounts(data['labelled']['total'], data['unlabelled'] + data['labelled']['total'])

      let accuracies = data['history'].map((step) => {
        return step['test']['acc']
      })

      if (accuracies.length > 0) {
        let maxAccuracy = Math.round(Math.max(...accuracies) * 100)
      } else {
        maxAccuracy = 0
      }
      setAccuracy(maxAccuracy)
    })
  }

  let setAccuracy = (maxAccuracy) => {
    $('#accuracy-text').html(`${maxAccuracy}%`)
  }

  let setLabelledCounts = (labelled, unlabelled) => {
    $('#labelled-counts-text').data('labelled', labelled)
    $('#labelled-counts-text').data('unlabelled', unlabelled)
    $('#labelled-counts-text').html(`${labelled} / ${unlabelled}`)
  }

  let incrementLabelledCounts = () => {
    let labelled = parseInt($('#labelled-counts-text').data('labelled'), 10)
    let unlabelled = parseInt($('#labelled-counts-text').data('unlabelled'), 10)
    $('#labelled-counts-text').data('labelled', labelled + 1)
    $('#labelled-counts-text').data('unlabelled', unlabelled)
    $('#labelled-counts-text').html(`${labelled + 1} / ${unlabelled}`)
  }

  let labellingDone = () => {
    $("#wrap").hide()
    $("#footer").hide()
    $("#done-page").show()
  }

  let getNextBatch = () => {
    $.get('/batch?prediction=' + false, (data) => {
      if (data['done']) {
        labellingDone()
        return
      }
      let batch = data['batch']
      itemsToLabel = itemsToLabel.concat(batch)
      if (currentIndex == 0) {
        showItem()
      }
    })
  }

  let submitJudgement = (label) => {
    let image = currentItem()
    let path = image.path

    $.post('/judgements', {id: path, label: label}, (response) => {
      if ('error' in response) {
        toastr.error(response['error'], 'Error!')
        return
      }
      showNextItem()
      incrementLabelledCounts()
      if (currentIndex + 3 > itemsToLabel.length) {
        getNextBatch()
        getTrainingHistory()
      }
    })
  }

  let showNextItem = () => {
    currentIndex += 1
    showItem()
  }

  // Kick everything off
  getNextBatch()
  getTrainingHistory()

  $(".judgement-button").click((el) => {
    let label = $(el.target).data("judgement")
    submitJudgement(label)
  })
})
