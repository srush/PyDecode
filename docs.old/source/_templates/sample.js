var w = 1500,
    h = 1500,
    fill = d3.scale.category20();

var vis = d3.select("#chart")
  .append("svg:svg")
    .attr("width", w)
    .attr("height", h);

d3.json("/notebooks/force.json", function(json) {
  // var force = d3.layout.tree()
  //     .nodes(json.nodes)
  //     .links(json.links)
  //     .size([w, h])
  //     .start();
  nodes = json.nodes

  function ypos(n) {
      return nodes[n].pos.split(",")[1];
  }

  function xpos(n) {
      console.log(n);
      return nodes[n].pos.split(",")[0];
  }

  var link = vis.selectAll("line.link")
      .data(json.links)
      .enter().append("svg:line")
      .attr("class", "link")
      .style("stroke-width", function(d) { return d.penwidth; })
      .style("stroke", function(d) { return d.color; })
      .attr("x1", function(d) { return xpos(d.source); })
      .attr("y1", function(d) { return ypos(d.source); })
      .attr("x2", function(d) { return xpos(d.target); })
      .attr("y2", function(d) { return ypos(d.target); });

  var node_group = vis.selectAll("circle.node")
      .data(json.nodes)
      .enter().append("g");
  node_group.attr("transform", function(d) { 
      return "translate(" + (d.pos.split(",")[0]) + " " + (d.pos.split(",")[1]) + ")"; });

  node_group
        .append("text")
        .text(function(d) { return d.label; } )

  node_group.filter(function(d) { return d.image != null; })
        .append("image")
        .attr("class", "node")
        .attr("xlink:href", function(d) { return "/notebooks/" + d.image; })
        .attr("anchor", "center")
        .attr("transform", "translate(-30, -20)")
        .attr("width", 60).attr("height", 40);
   
  node_group.filter(function(d) { return d.image == null; })
        .append("circle")
        .attr("class", "node")
        .attr("r", "5")
        .style("fill", function(d) { return d.color; })
      // .style("fill", function(d) { return d.color; })
      //.style("background-image", function(d) { return "url(\"/notebooks/" + d.image + "\")"; })
      //.attr("r", 100);
      // 
      // .call(force.drag);

  // node.append("svg:title")
  //     .text(function(d) { return d.name; });

  // vis.style("opacity", 1e-6)
  //   .transition()
  //     .duration(1000)
  //     .style("opacity", 1);

  // force.on("tick", function() {
  //   link.attr("x1", function(d) { return d.source.x; })
  //       .attr("y1", function(d) { return d.source.y; })
  //       .attr("x2", function(d) { return d.target.x; })
  //       .attr("y2", function(d) { return d.target.y; });

  //   node.attr("cx", function(d) { return d.x; })
  //       .attr("cy", function(d) { return d.y; });
  // });
});
