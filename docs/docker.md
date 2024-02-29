### refer below to make use of the docker
- had not used the .dockerignore and hence that is why that big build, fixable

```
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker images
REPOSITORY             TAG       IMAGE ID       CREATED         SIZE
vershachi-unlearning   latest    3bcf72c0120d   8 minutes ago   12.8GB
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker run -itd --name vershachi-1 -p 5555:80 vershachi-unlearning:latest
5f7d2e5f8a36c991f96568b74a3d631f33642ef8b4c25bd02d1cc4052690bbb2
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker ps -a
CONTAINER ID   IMAGE                         COMMAND           CREATED          STATUS                      PORTS     NAMES
5f7d2e5f8a36   vershachi-unlearning:latest   "python run.py"   14 seconds ago   Exited (0) 10 seconds ago             vershachi-1
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker run
"docker run" requires at least 1 argument.
See 'docker run --help'.

Usage:  docker run [OPTIONS] IMAGE [COMMAND] [ARG...]

Create and run a new container from an image
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker logs vershachi-1
Hello! This is the Vershachi Unlearning Framework.
Performing sanity checks...
Module 'tqdm' is installed.
Module 'torch' is installed.
Module 'scikit-learn' is installed.
Module 'black' is installed.
Module 'torchvision' is installed.
Module 'xgboost' is installed.
Module 'numpy' is installed.
Module 'pandas' is installed.
All required modules are installed.
Vershachi Unlearning Framework version 0.1
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker exec -it 5f7d2e5f8a36 /bin/bash
Error response from daemon: Container 5f7d2e5f8a36c991f96568b74a3d631f33642ef8b4c25bd02d1cc4052690bbb2 is not running
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker start 5f7d2e5f8a36
5f7d2e5f8a36
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker rm 5f7d2e5f8a36
5f7d2e5f8a36
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker rmi 3bcf72c0120d
Untagged: vershachi-unlearning:latest
Deleted: sha256:3bcf72c0120d0af1d4406f45ab977053700145e455862c19a4d74cddce63455a
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker container prune
WARNING! This will remove all stopped containers.
Are you sure you want to continue? [y/N]
Total reclaimed space: 0B
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker image prune
WARNING! This will remove all dangling images.
Are you sure you want to continue? [y/N]
Total reclaimed space: 0B
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker system prune
WARNING! This will remove:
  - all stopped containers
  - all networks not used by at least one container
  - all dangling images
  - all dangling build cache

Are you sure you want to continue? [y/N]
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker system df
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          0         0         0B        0B
Containers      0         0         0B        0B
Local Volumes   1         0         206.9MB   206.9MB (100%)
Build Cache     14        0         14.52GB   14.52GB
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ docker builder prune --all --force
ID                                              RECLAIMABLE     SIZE            LAST ACCESSED
yogjg3rrirvmab64jbigj3fwb                       true            9.011GB         23 minutes ago
zzi9f5zce2hajqwlkofnlnqmd*                      true    464B            23 minutes ago
46zrivlvuesqsbgn1qt6p5b5l*                      true    0B              23 minutes ago
dnpktqzvc1avbx7at62o1p6gf*                      true    2.752GB         23 minutes ago
b3dexxkm68rgrvttvsy4kqzb7                       true    2.752GB         23 minutes ago
rzyxh0k74aa8v3pzrl6m5gq6a                       true    0B              23 minutes ago
r8vkg9lgakxpxle8r7r01691k                       true    0B              23 minutes ago
o6i5dh7vb9msms41336je11h0                       true    0B              About an hour ago
yb5gdmgwpjt2xin3mucwvmi5e                       true    0B              About an hour ago
ivu40w8n4dcp2cvv3jggq3vp8                       true    0B              About an hour ago
zei57xbto9s91c0yblacfao0i                       true    0B              About an hour ago
azojv3j8qikxbtlqcr56w0ltd                       true    0B              About an hour ago
k9tobzjpj43lm4rtpj6m5f481                       true    0B              About an hour ago
zr80j8uqlquh2ln6xlo3r71i7                       true    0B              About an hour ago
Total:  14.52GB
(venv) shahab in C:\dev\vershachi-unlearning on main ● ?2 ~2 λ
```