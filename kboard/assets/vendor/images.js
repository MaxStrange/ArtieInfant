var chart;
var length = 0;

function requestData(metric) {
  $.ajax({
    url: "/data",
    data: {"x": length, "metric": metric},
    success: function(y) {
      length += 1;
      var x = (new Date()).getTime(), // current time
      series = chart.series[0],
      shift = series.data.length > 20;
      series.addPoint([x, y], true, shift);
      setTimeout(function() {requestData("acc")}, 1000);
    },
    cache: false
  });
}

$(document).ready(function() {
  chart = new Highcharts.Chart({
    chart: {
      renderTo: "container",
      defaultSeriesType: "spline",
      events: {
        load: function() {
          requestData("acc");
        }
      }
    },
    title: {
      text: "Live Random Data"
    },
    xAxis: {
      type: "datetime",
      tickPixelInterval: 150,
      maxZoom: 20 * 1000
    },
    yAxis: {
      minPadding: 0.2,
      maxPadding: 0.2,
      title: {
        text: "Value",
        margin: 80
      }
    },
    series: [{
      name: "Random Data",
      data: []
    }]
  });
});
//var frequency = 2000;//ms
//var interval = 0;
//
//function startLoop() {
//    if (interval > 0)
//        clearInterval(interval);
//    interval = setInterval("updateImages()", frequency);
//}
//
//function updateImages() {
//    // Get all the images in the image div
//    var images = document.getElementById("images_div").getElementsByTagName("img");
//    var now = new Date();
//
//    var add_to_width = 0;
//    // TODO: Get `add_to_width` from JSON.
//    $.getJSON("http://localhost:4000/data", function(result) {
//        console.log(result);
//        add_to_width = result;
//    });
//
//    // Update all image srcs
//    for (i = 0; i < images.length; i++) {
//        //images[i].src = images[i].src + "?" + now.getTime();
//        images[i].width += add_to_width;
//    }
//    timeoutID = setTimeout("updateImages()", 60000);
//}
//
//$(document).ready(function () {
//    Highcharts.setOptions({
//      global: {
//        useUTC: false
//      }
//    });
//
//    Highcharts.chart('container', {
//      chart: {
//        type: 'spline',
//        animation: Highcharts.svg, // don't animate in old IE
//        marginRight: 10,
//        events: {
//          load: function () {
//
//            // set up the updating of the chart each second
//            var series = this.series[0];
//            setInterval(function () {
//              var x = (new Date()).getTime(), // current time
//              y = Math.random();
//              series.addPoint([x, y], true, true);
//            }, 1000);
//          }
//        }
//      },
//      title: {
//        text: 'Live random data'
//      },
//      xAxis: {
//        type: 'datetime',
//        tickPixelInterval: 150
//      },
//      yAxis: {
//        title: {
//          text: 'Value'
//        },
//        plotLines: [{
//          value: 0,
//          width: 1,
//          color: '#808080'
//        }]
//      },
//      tooltip: {
//        formatter: function () {
//          return '<b>' + this.series.name + '</b><br/>' +
//          Highcharts.dateFormat('%Y-%m-%d %H:%M:%S', this.x) + '<br/>' +
//          Highcharts.numberFormat(this.y, 2);
//        }
//    },
//    legend: {
//      enabled: false
//    },
//    exporting: {
//      enabled: false
//    },
//    series: [{
//      name: 'Random data',
//      data: (function () {
//        // generate an array of random data
//        var data = [],
//        time = (new Date()).getTime(),
//        i;
//
//        for (i = -19; i <= 0; i += 1) {
//          data.push({
//            x: time + i * 1000,
//            y: Math.random()
//          });
//        }
//        return data;
//      }())
//    }]
//  });
//});
