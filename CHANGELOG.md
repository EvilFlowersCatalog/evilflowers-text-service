# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- The default `BROKER_VISIBILITY_TIMEOUT` is now 43200 s (12 h), up from 7200 s. Every consumer
  of a Redis db restores all unacknowledged messages older than its own timeout, in any queue. On
  dev, `text` shares db 7 with the catalog, analyzer, image, equation and lcpencrypt, and an equation
  pass has taken over 2 h, so 2 h caused redeliveries. 12 h matches the catalog and the analyzer.
  A task lost to a killed pod now comes back after up to 12 h instead of 2 h. The variable still
  overrides the default. ([#9](https://github.com/EvilFlowersCatalog/evilflowers-text-service/issues/9))
