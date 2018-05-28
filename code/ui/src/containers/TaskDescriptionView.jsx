import React from 'react';

class TaskDescriptionView extends React.Component {
  render() {
    return (
      <div>
        <h2 dangerouslySetInnerHTML={{__html: this.props.title}} className="center">
        </h2>
        <div dangerouslySetInnerHTML={{__html: this.props.description}} className="center">
        </div>
      </div>
    );
  }
}

export default TaskDescriptionView;
