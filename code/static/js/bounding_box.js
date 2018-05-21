$(document).ready(() => {

  var canvas = document.getElementById("canvas");
  var ctx = canvas.getContext("2d");
  var img = new Image;
  img.onload = function(){
    ctx.drawImage(img, 0, 0, img.width,    img.height,     // source rectangle
                   0, 0, canvas.width, canvas.height); // Or at whatever offset you like
  };
  img.src = 'https://png.pngtree.com/element_pic/16/11/18/b82016401753f3bfd306ac8ce82c82c0.jpg'

  // style the context
  ctx.lineWidth = 3;

  // calculate where the canvas is on the window
  // (used to help calculate mouseX/mouseY)
  var $canvas = $("#canvas");
  var canvasOffset = $canvas.offset();
  var offsetX = canvasOffset.left;
  var offsetY = canvasOffset.top;
  var scrollX = $canvas.scrollLeft();
  var scrollY = $canvas.scrollTop();

  // this flage is true when the user is dragging the mouse
  var isDown = false;

  // these vars will hold the starting mouse position
  var startX;
  var startY;
  var boxes = []; // TODO: Need to add colors to the objects added
  var currentColor = '#eeeeee';
  var currentClass = '';

  $(document).keydown((e) => {
    if (e.keyCode === 8) {
      e.preventDefault();
      boxes.splice(-1,1);
      draw();
    }
  });

  function handleMouseDown(e) {
      e.preventDefault();
      e.stopPropagation();

      // save the starting x/y of the rectangle
      startX = parseInt(e.clientX - offsetX);
      startY = parseInt(e.clientY - offsetY);

      // set a flag indicating the drag has begun
      isDown = true;
  }

  function handleMouseUp(e) {
      e.preventDefault();
      e.stopPropagation();

      // the drag is over, clear the dragging flag
      isDown = false;
      mouseX = parseInt(e.clientX - offsetX);
      mouseY = parseInt(e.clientY - offsetY);
      var width = mouseX - startX;
      var height = mouseY - startY;
      // Get currently selected detection-button
      boxes.push({startX, startY, width, height, color: ctx.strokeStyle});
  }

  function handleMouseOut(e) {
      e.preventDefault();
      e.stopPropagation();

      // the drag is over, clear the dragging flag
      isDown = false;
  }
  function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, canvas.width, canvas.height); // Or at whatever offset you like
      for (var i = 0; i < boxes.length; i++) {
        var boxStartX = boxes[i].startX;
        var boxStartY = boxes[i].startY;
        var boxWidth = boxes[i].width;
        var boxHeight = boxes[i].height;
        var color = boxes[i].color;
        drawBox(ctx, boxStartX, boxStartY, boxWidth, boxHeight, color);
      }
  }

  function handleMouseMove(e) {
      e.preventDefault();
      e.stopPropagation();

      // if we're not dragging, just return
      if (!isDown) {
          return;
      }

      // get the current mouse position
      mouseX = parseInt(e.clientX - offsetX);
      mouseY = parseInt(e.clientY - offsetY);

      // Put your mousemove stuff here

      // clear the canvas
      draw();
      // calculate the rectangle width/height based
      // on starting vs current mouse position
      var width = mouseX - startX;
      var height = mouseY - startY;
      // draw a new rect from the start position
      // to the current mouse position
      drawBox(ctx, startX, startY, width, height, currentColor);
  }
  function drawBox(ctx, startX, startY, width, height, color) {
      ctx.strokeStyle = color;
      ctx.strokeRect(startX, startY, width, height);
  }

  // listen for mouse events
  $("#canvas").mousedown(function (e) {
      handleMouseDown(e);
  });
  $("#canvas").mousemove(function (e) {
      handleMouseMove(e);
  });
  $("#canvas").mouseup(function (e) {
      handleMouseUp(e);
  });
  $("#canvas").mouseout(function (e) {
      handleMouseOut(e);
  });

  let detectionClasses = window.detectionClasses;
  let selectedClass = null;
  $('.detection-button').click((el) => {
    let buttons = $('.detection-button')
    for (let i = 0; i < buttons.length; i++) {
      let button = buttons[i];
      let color = $(this).data('color');
      $(button).removeAttr('style');
      $(button).attr('style', `border: 1px solid ${color};`);
    }

    let color = $(el.target).data('color');
    $(el.target).attr('style', `background: ${color};`);
    selectedClass = $(el.target).data('detection-class')
    currentColor = color;
    currentClass = selectedClass;
  });

  function submitBoundBox() {
  }
});
