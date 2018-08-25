import React from 'react';

class BoundingBoxDemoView extends React.Component {
  updateCanvas() {
    const canvas = this.refs.canvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let img = new Image();
    img.src = `/images?image_path=${this.props.url}`;
    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
    // Draw bounding boxes
    this.props.boundingBoxes.forEach((boundingBox) => {
      this.drawBoxWithClassName(ctx, boundingBox);
    });
  }

  drawBoxWithClassName(ctx, boundingBox) {
    let startX = boundingBox['startX'];
    let startY = boundingBox['startY'];
    let width = boundingBox['width'];
    let height = boundingBox['height'];
    let classString = boundingBox['class'];
    ctx.strokeStyle = '#1abc9c';
    ctx.lineWidth = 5;
    ctx.strokeRect(startX, startY, width, height);
    ctx.font="20px Georgia";
    ctx.fillText(classString, startX, startY - 30);
  }

  componentDidMount() {
    this.updateCanvas()
  }

  componentDidUpdate() {
    this.updateCanvas()
  }

  render() {
    return (
      <canvas ref="canvas" id="canvas" width="750" height="500">
        Your browser does not support the HTML5 canvas tag.
      </canvas>
    );
  }
}

export default BoundingBoxDemoView;
