/*
 * @(#)FileUtils.java   1.0   May 15, 2019
 *
 * Copyright (c) 2019- University of Konstanz.
 *
 * This software is the proprietary information of the above-mentioned institutions.
 * Use is subject to license terms. Please refer to the included copyright notice.
 */
package kn.uni.dbis.neo4j.eval.util;

import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Utilities for creating, deleting, copying files and directories.
 *
 * @author Manuel Hotz &lt;manuel.hotz@uni-konstanz.de&gt;
 * @author Fabian Klopfer &lt;fabian.klopfer@uni-konstanz.de&gt;
 */
public final class FileUtils {
  /**
   * Feature flag for linux file clones on BTRFS.
   */
  private static final boolean LINUX_CLONE = false;
  /**
   * stores the users OS name in upper case letters for comparison.
   */
  private static final String OS_NAME = System.getProperty("os.name").toUpperCase();

  /**
   * Clones the given file or folder into the destination folder.
   * A call {@code clone(/tmp/src.test, /tmp/dest)} will result in a COW folder {@code /tmp/dest/src.test}.
   *
   * @param src        source folder/file to clone
   * @param destFolder destination folder to put file/folder into
   * @return new file/folder path
   * @throws IOException if the copying process fails
   */
  public Path clone(final Path src, final Path destFolder) throws IOException {
    final Path destPath = Paths.get(destFolder.toAbsolutePath().toString(), src.getFileName().toString());

    // Windows does not support Copy-On-Write on the typical user install
    if (this.is(OS.WINDOWS)) {
      System.err.println("[WARNING] Cannot use Copy-On-Write");
      this.copyTo(src, destPath);
      return destPath;
    }

    this.cloneTo(src, destPath);
    return destPath;
  }

  /**
   * Clones a folder to another directory.
   *
   * @param src      directory to be copied
   * @param destPath destination path
   * @throws IOException throws if copy process was unsucessful
   */
  private void cloneTo(final Path src, final Path destPath) throws IOException {
    // copy-on-write files
    final ProcessBuilder proc = new ProcessBuilder();
    if (this.is(OS.MAC)) {
      // cp -c uses clonefile (APFS on MacOS)
      proc.command("cp",
          "-c",
          "-f",
          src.toFile().isDirectory() ? "-R" : " ",
          src.toAbsolutePath().toString(),
          destPath.toString());
      // assume we are running on Linux
    } else if (this.is(OS.LINUX)) {
      // cp --reflink uses reflink (BTRFS on Linux)
      proc.command("cp" + (LINUX_CLONE ? "--reflink" : ""),
          "--force",
          src.toFile().isDirectory() ? "--recursive" : "--no-target-directory",
          src.toAbsolutePath().toString(),
          destPath.toAbsolutePath().getParent().toString());
    } else {
      throw new IllegalStateException("Unsupported platform: " + OS_NAME);
    }
    proc.directory(src.getParent().toAbsolutePath().toFile());
    proc.inheritIO();
    final int code;
    try {
      code = proc.start().waitFor();
    } catch (final InterruptedException e) {
      throw new IOException(e);
    }
    if (code != 0) {
      throw new IOException(String.format("Clone %s failed with exit code %d", proc.command(), code));
    }
  }

  /**
   * Fallback recursive copy method.
   *
   * @param src    source folder
   * @param target target folder
   * @throws IOException io exception
   */
  private void copyTo(final Path src, final Path target) throws IOException {
    final List<Path> sources = Files.walk(src).collect(Collectors.toList());
    final List<Path> dests = sources.stream().map(src::relativize).map(target::resolve).collect(Collectors.toList());
    for (int i = 0; i < sources.size(); i++) {
      final Path f = sources.get(i);
      final Path d = dests.get(i);
      System.out.println("Copying " + f);
      System.out.println("to " + d);
      Files.copy(f, d, StandardCopyOption.REPLACE_EXISTING);
    }
  }

  /**
   * Deletes Path recursively on exit.
   *
   * @param path path to delete
   * @throws IOException throws if deletion was unsuccessful
   */
  public void recursiveDeleteOnExit(final Path path) throws IOException {
    Files.walkFileTree(path, new SimpleFileVisitor<Path>() {
      @Override
      public FileVisitResult visitFile(final Path file,
                                       @SuppressWarnings("unused") final BasicFileAttributes attrs) {
        file.toFile().deleteOnExit();
        return FileVisitResult.CONTINUE;
      }

      @Override
      public FileVisitResult preVisitDirectory(final Path dir,
                                               @SuppressWarnings("unused") final BasicFileAttributes attrs) {
        dir.toFile().deleteOnExit();
        return FileVisitResult.CONTINUE;
      }
    });
  }

  /**
   * used to handle different OS types like if (Platform.is(Platform.LINUX)).
   *
   * @param os The Operating system to compare to.
   * @return true if the users os is the os specified in the parameter
   */
  private boolean is(final OS os) {
    return OS_NAME.startsWith(os.name());
  }

  /**
   * Enum specifying the most frequent operating system names.
   */
  public enum OS {
    /**
     * MAC OS.
     */
    MAC,
    /**
     * Linux.
     */
    LINUX,
    /**
     * Windows.
     */
    WINDOWS,
    /**
     * Everything else.
     */
    UNSUPPORTED
  }
}
