import React from 'react';

class JSONTaskView extends React.Component {
  render() {
    let json = JSON.parse(this.props.currentItem['text']);
    let template = this.props.task.template;

    if (template) {
      Object.keys(json).forEach(function(key) {
        let value = json[key];
        template = template.replace(`{{${key}}}`, value);
      });
    }
    return (
      <div dangerouslySetInnerHTML={{__html: template}} className="center">
      </div>
    );
  }
}

export default JSONTaskView;
