#!/usr/bin/perl
# macOS has no setsid(1): start the command in its own session/process group (same PID, so the recorded pid is the group leader to signal).
use POSIX qw(setsid);
setsid();
exec @ARGV or die "exec failed: $!";
