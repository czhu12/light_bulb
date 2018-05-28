import { connect } from 'react-redux';
import React from 'react';

class DemoApp extends React.Component {
  render() {
    return (
      <div>
        Label App
      </div>
    );
  }
}

const mapStateToProps = state => ({
  state: state
});


const mapDispatchToProps = dispatch => ({
});

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(DemoApp);
