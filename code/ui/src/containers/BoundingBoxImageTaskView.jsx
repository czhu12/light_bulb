import { connect } from 'react-redux';
import React from 'react';
import { submitJudgement } from '../actions';
import { CLASSIFICATION_COLORS } from './LabelClassificationView';

class BoundingBoxImageTaskView extends React.Component {
  constructor(props) {
    super(props);
    let img = new Image();
    this.state = {
      boxes: [],
      isDown: false,
      img,
    }
  }

  drawBox(ctx, startX, startY, width, height, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 5;
    ctx.strokeRect(startX, startY, width, height);
  }

  getFileName(src) {
    let split = src.split('/');
    return split[split.length - 1];
  }

  updateCanvas() {
    const canvas = this.refs.canvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let boxes = this.state.boxes;
    let isDown = this.state.isDown;
    let imgSrc = `/images?image_path=${this.props.currentItem['path']}`
    let fileName = this.getFileName(imgSrc);
    let existingFileName = this.getFileName(this.state.img.src);

    if (fileName !== existingFileName) {
      this.state.img.src = imgSrc;
      this.state.img.onload = () => {
        ctx.drawImage(this.state.img, 0, 0, canvas.width, canvas.height);
      }
    } else {
      ctx.drawImage(this.state.img, 0, 0, canvas.width, canvas.height);
    }

    for (let i = 0; i < boxes.length; i++) {
      let boxStartX = boxes[i].startX;
      let boxStartY = boxes[i].startY;
      let boxWidth = boxes[i].width;
      let boxHeight = boxes[i].height;
      let color = boxes[i].color;
      this.drawBox(ctx, boxStartX, boxStartY, boxWidth, boxHeight, color);
    }

    let canvasOffset = canvas.getBoundingClientRect();
    let offsetX = canvasOffset.left;
    let offsetY = canvasOffset.top;
    if (isDown) {
      let width = this.state.mouseX - this.state.startX;
      let height = this.state.mouseY - this.state.startY;
      this.drawBox(
        ctx,
        this.state.startX - offsetX,
        this.state.startY - offsetY,
        width,
        height,
        this._currentColor(),
      );
    }
  }

  _currentColor() {
    console.log(this.props);
    let colorIndex = this.props.classes.indexOf(this.props.currentBoundingBoxClass);
    return CLASSIFICATION_COLORS[colorIndex];
  }

  onMouseDown(e) {
    if (!this.state) return;
    let startX = parseInt(e.clientX, 10);
    let startY = parseInt(e.clientY, 10);
    let state = {
      ...this.state,
      startX,
      startY,
      isDown: true,
    };
    this.setState(state);
  }

  onMouseUp(e) {
    if (!this.state) return;
    let mouseX = parseInt(e.clientX, 10);
    let mouseY = parseInt(e.clientY, 10);
    let width = mouseX - this.state.startX;
    let height = mouseY - this.state.startY;

    let boxes = this.state.boxes.slice();
    const canvas = this.refs.canvas;
    let canvasOffset = canvas.getBoundingClientRect();
    let offsetX = canvasOffset.left;
    let offsetY = canvasOffset.top;

    boxes.push({
      startX: this.state.startX - offsetX,
      startY: this.state.startY - offsetY,
      color: this._currentColor(),
      width: width,
      height: height,
      currentClass: this.props.currentBoundingBoxClass,
    })

    this.setState({
      ...this.state,
      boxes: boxes,
      mouseX: mouseX,
      mouseY: mouseY,
      isDown: false,
    })
  }

  onMouseMove(e) {
    if (!this.state) return;
    if (!this.state.isDown) {
      return;
    }
    let mouseX = parseInt(e.clientX, 10);
    let mouseY = parseInt(e.clientY, 10);
    this.setState({
      ...this.state,
      mouseX,
      mouseY,
    })
  }

  onMouseOut(e) {
    if (!this.state) return;
    this.setState({
      ...this.state,
      isDown: false,
    })
  }

  componentDidUpdate() {
    this.updateCanvas();
  }

  buildJudgement() {
    const canvas = this.refs.canvas;
    let canvasOffset = canvas.getBoundingClientRect();
    let offsetX = canvasOffset.left;
    let offsetY = canvasOffset.top;

    return JSON.stringify(this.state.boxes.map((box) => {
      return {
        ...box,
        startX: box.startX - offsetX,
        startY: box.startY - offsetY,
      }
    }));
  }

  onKeyPress(e) {
    if (e.key === 'Enter') {
      this.props.submitJudgement(JSON.stringify(this.state.boxes));
      this.setState({
        ...this.state,
        isDown: false,
        boxes: [],
      })
    }
  }

  onKeyDown(e) {
    if (e.key === "Backspace") {
      let boxes = this.state.boxes.slice(0);
      boxes.splice(-1,1)
      this.setState({
        ...this.state,
        isDown: false,
        boxes,
      });
    }
  }

  componentDidUnmount() {
    window.removeEventListener("keypress", this.onKeyPress.bind(this));
    window.removeEventListener("keydown", this.onKeyDown.bind(this));
  }

  componentDidMount() {
    this.updateCanvas();
    window.addEventListener("keypress", this.onKeyPress.bind(this));
    window.addEventListener("keydown", this.onKeyDown.bind(this));
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
  classes: state.task.classes,
});


const mapDispatchToProps = dispatch => ({
  submitJudgement: (serializedBoxes) => {
    dispatch(submitJudgement(serializedBoxes));
  },
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(BoundingBoxImageTaskView);
