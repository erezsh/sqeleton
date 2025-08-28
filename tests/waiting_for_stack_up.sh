#!/bin/bash

if [ -n "$VERTICA_URI" ]
    then
        echo "Check Vertica DB running..."
        while true
        do
            if docker logs dd-vertica | tail -n 100 | grep -q -i "vertica is now running"
            then
               echo "Vertica DB is ready";
               break;
            else
               echo "Waiting for Vertica DB starting...";
               sleep 10;
            fi
        done
fi

if [ -n "${DREMIO_URI}" ]
    then
        echo "Check Dremio DB running..."
        docker ps

        while true
        do
            if docker logs dd-dremio | tail -n 1000 | grep -q -i "Dremio Daemon Started as master"
            then
               echo "Dremio DB is ready";
               break;
            else
               echo "Waiting for Dremio DB starting...";
               sleep 10;
            fi
        done

        echo "Create first user in Dremio DB..."
        curl 'http://localhost:9047/apiv2/bootstrap/firstuser' -X PUT \
          -H 'Authorization: _dremionull' -H 'Content-Type: application/json' \
          --data-binary \
          '{"userName":"dremio","firstName":"your_fname","lastName":"your_lname","email":"your_email","password":"dremio123"}'

        echo "Dremio DB is ready";
fi