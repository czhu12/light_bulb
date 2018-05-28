import React from 'react';

class TaskDescriptionView extends React.Component {
  render() {
    return (
      <div>
        <h2 className="center">
          {this.props.title}
        </h2>
        <div className="center">
          {this.props.description}
        </div>
      </div>
    );
  }
}

export default TaskDescriptionView;
