import React from 'react';

const initialState = {

}

class BoundingBoxImageTaskView extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isDown: false,
      startX: 0,
      startY: 0,
      boxes: [],
      currentColor: '',
      currentClass: '',
    }
  }

  updateCanvas() {
    const canvas = this.refs.canvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let img = new Image();
    img.src = `/images?image_path=${this.props.currentItem['path']}`;
    img.onload = function() {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }

    for (let i = 0; i < this.state.boxes.length; i++) {
      let boxStartX = boxes[i].startX;
      let boxStartY = boxes[i].startY;
      let boxWidth = boxes[i].width;
      let boxHeight = boxes[i].height;
      let color = boxes[i].color;
      drawBox(ctx, boxStartX, boxStartY, boxWidth, boxHeight, color);
    }
  }

  onMouseDown(e) {
    this.setState({
      ...this.state,
      startX: parseInt(e.clientX - this.state.offsetX),
      startY: parseInt(e.clientY - this.state.offsetY),
      isDown: true,
    })
  }

  onMouseMove(e) {
    if (!this.state.isDown) {
      return;
    }
    this.setState({
      ...this.state,
      mouseX: parseInt(e.clientX - this.state.offsetX);
      mouseY: parseInt(e.clientY - this.state.offsetY);

    })
  }

  onMouseUp(e) {
  }

  onMouseOut(e) {
  }

  componentDidUpdate() {
    this.updateCanvas();
  }

  componentDidMount() {
    const canvas = this.refs.canvas;
    const canvasOffset = canvas.offset();
    this.setState({
      ...this.state,
      offsetX: canvasOffset.left,
      offsetY: canvasOffset.top,
      scrollX: canvas.scrollLeft,
      scrollY: canvas.scrollTop,
    })

    this.updateCanvas();
  }

  render() {
    return (
      <canvas
        onMouseDown={this.onMouseDown}
        onMouseMove={this.onMouseMove}
        onMouseUp={this.onMouseUp}
        onMouseOut={this.onMouseOut}
        ref="canvas"
        id="canvas"
        width="750"
        height="500"
      >
        Your browser does not support the HTML5 canvas tag.
      </canvas>
    );
  }
}

export default BoundingBoxImageTaskView;
