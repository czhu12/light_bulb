import React from 'react';

class ImageTaskBatchView extends React.Component {
  render() {
    let images = this.props.items.map((item) => {
      let path = item['path'];
      let src = `/images?image_path=${path}`;

      return (
        <li className="list-group-item">
          <img className="img-thumbnail float-left" src={src} />
        </li>
      );
    });

    return (
      <ul className="list-group">
        {images}
      </ul>
    );
  }
}

export default ImageTaskBatchView;
