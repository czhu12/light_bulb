import { connect } from 'react-redux';
import React from 'react';

class BoundingBoxImageTaskView extends React.Component {
  //drawBox(ctx, startX, startY, width, height, color) {
  //  //console.log(ctx);
  //  //console.log(startX);
  //  //console.log(startY);
  //  //console.log(width);
  //  //console.log(height);
  //  //console.log(color);
  //  ctx.strokeStyle = color;
  //  ctx.strokeRect(startX, startY, width, height);
  //}

  //updateCanvas() {
  //  const canvas = this.refs.canvas;
  //  const ctx = canvas.getContext("2d");
  //  ctx.clearRect(0, 0, canvas.width, canvas.height);

  //  let boxes = this.state.boxes;
  //  let isDown = this.state.isDown;
  //  //if (imgSrc !== img.src) {
  //  //  boxes = [];
  //  //  isDown = false;
  //  //  this.setState({
  //  //    ...this.state,
  //  //    boxes,
  //  //    isDown,
  //  //  })
  //  //}

  //  ctx.drawImage(this.state.img, 0, 0, canvas.width, canvas.height);
  //  for (let i = 0; i < boxes.length; i++) {
  //    let boxStartX = boxes[i].startX;
  //    let boxStartY = boxes[i].startY;
  //    let boxWidth = boxes[i].width;
  //    let boxHeight = boxes[i].height;
  //    let color = boxes[i].color;
  //    this.drawBox(ctx, boxStartX, boxStartY, boxWidth, boxHeight, color);
  //  }

  //  // Draw the current box
  //  if (isDown) {
  //    let width = this.state.mouseX - this.state.startX;
  //    let height = this.state.mouseY - this.state.startY;
  //    this.drawBox(
  //      ctx,
  //      this.state.startX,
  //      this.state.startY,
  //      width,
  //      height,
  //      this.state.currentColor,
  //    );
  //  }
  //}

  onMouseDown(e) {
  //  if (!this.state) return;
  //  let state = {
  //    ...this.state,
  //    startX: parseInt(e.clientX - this.state.offsetX, 10),
  //    startY: parseInt(e.clientY - this.state.offsetY, 10),
  //    isDown: true,
  //  };
  //  console.log(state);
  //  this.setState(state);
  }

  onMouseMove(e) {
  //  if (!this.state) return;
  //  if (!this.state.isDown) {
  //    return;
  //  }
  //  this.setState({
  //    ...this.state,
  //    mouseX: parseInt(e.clientX - this.state.offsetX, 10),
  //    mouseY: parseInt(e.clientY - this.state.offsetY, 10),
  //  })
  }

  onMouseUp(e) {
  //  if (!this.state) return;
  //  let mouseX = parseInt(e.clientX - this.state.offsetX, 10);
  //  let mouseY = parseInt(e.clientX - this.state.offsetY, 10);
  //  let width = mouseX - this.state.startX;
  //  let height = mouseY - this.state.startY;

  //  let boxes = this.state.boxes.slice();
  //  boxes.push({
  //    startX: this.state.startX,
  //    startY: this.state.startY,
  //    currentColor: this.state.currentColor,
  //    width: width,
  //    height: height,
  //    currentClass: this.props.currentBoundingBoxClass,
  //  })

  //  this.setState({
  //    ...this.state,
  //    boxes: boxes,
  //    mouseX: mouseX,
  //    mouseY: mouseY,
  //    isDown: false,
  //  })
  }

  onMouseOut(e) {
  //  if (!this.state) return;
  //  this.setState({
  //    ...this.state,
  //    isDown: false,
  //  })
  }

  //componentDidUpdate() {
  //  this.updateCanvas();
  //}

  //onJudgement() {
  //  this.props.submitJudgement(JSON.stringify(this.state.boxes));
  //}

  //componentDidUnmount() {
  //  window.removeEventListener("keypress", this.onJudgement);
  //}

  componentDidMount() {
    console.log("HELLO WORLD");
    const canvas = this.refs.canvas;
    const ctx = canvas.getContext("2d");
    console.log(canvas);
    console.log(ctx);
    console.log(canvas.offset());
    const canvasOffset = canvas.offset();
    let img = new Image();
    let imgSrc = `/images?image_path=${this.props.currentItem['path']}`
    img.src = imgSrc;
    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }

    this.setState({
      scrollX: canvas.scrollLeft,
      scrollY: canvas.scrollTop,
      isDown: false,
      startX: 0,
      startY: 0,
      boxes: [],
      currentColor: '#2ecc71',
      currentClass: 'player',
      offsetX: canvasOffset.left,
      offsetY: canvasOffset.top,
      img: img,
    })

  //  this.updateCanvas();
  }

  render() {
    return (
      <canvas
        onMouseDown={this.onMouseDown.bind(this)}
        onMouseMove={this.onMouseMove.bind(this)}
        onMouseUp={this.onMouseUp.bind(this)}
        onMouseOut={this.onMouseOut.bind(this)}
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

const mapStateToProps = state => ({
  currentBoundingBoxClass: state.items.currentBoundingBoxClass,
});


const mapDispatchToProps = dispatch => ({
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(BoundingBoxImageTaskView);
