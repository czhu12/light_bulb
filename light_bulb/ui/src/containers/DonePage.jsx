import React from 'react';

class DonePage extends React.Component {
  render() {
    if (this.props.done) {
      return (
        <div id="done-page">
          <div className="container center">
            <h2>You're Done!</h2>
            <img
              alt="You're Done!"
              className="img-responsive"
              src="https://www.hellomagazine.com/imagenes/the-buzz/2017052239157/Success-Kid-unrecognisable/0-207-858/sammy-t.jpg"
            />
          </div>
        </div>
      );
    } else {
      return null;
    }
  }
}

export default DonePage;
