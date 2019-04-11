package visualize;

import cluster.dendrogram.DendrogramNode;
import preprocess.DataObject;

import javax.swing.*;
import java.awt.*;
import java.util.List;

// TODO print
public class DendrogramJPanel<T extends DataObject> extends JPanel {
  private List<DendrogramNode<T>> tree;

  public DendrogramJPanel(List<DendrogramNode<T>> tree) {
    this.tree = tree;
  }

  @Override
  public void paintComponent(Graphics g) {
    int width = this.getParent().getWidth();
    int height = this.getParent().getHeight();
    int levelFactorX = 2;
    int levelFactorY = 1;
    DendrogramNode<T> node = tree.get(0);

    while (!node.isLeaf()) {
      g.drawOval(width / levelFactorX, 25 * levelFactorY, 5, 5);
      g.drawLine(width / levelFactorX, 25 * levelFactorY,
          width / (levelFactorX + 1),
          25 * levelFactorY + 1);
      g.drawLine(width / levelFactorX, 25 * levelFactorY,
          width / (levelFactorX + 1),
          25 * levelFactorY + 1);
    }

  }

}
