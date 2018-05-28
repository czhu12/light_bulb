import React from 'react';

class TextTaskView extends React.Component {
  render() {
    return (
      <div id="text-classification-text">
        {this.props.text}
      </div>
    );
  }
}

export default TextTaskView;
