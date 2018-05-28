import React from 'react';

class ImageTaskView extends React.Component {
  componentDidMount() {
    const canvas = this.refs.canvas;
    const ctx = canvas.getContext("2d");
    let img = new Image();
    img.src = `/images?image_path=${this.props.currentItem['path']}`;
    img.onload = function() {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
  }

  render() {
    return (
      <canvas ref="canvas" id="canvas" width="750" height="500">
        Your browser does not support the HTML5 canvas tag.
      </canvas>
    );
  }
}

export default ImageTaskView;
